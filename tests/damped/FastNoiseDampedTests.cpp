#include "GradientProbe.h"

#include <FastNoise/FastNoise.h>
#include <FastNoise/Metadata.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
    using FastNoise::SmartNode;
    using Formula = GradientProbe::Formula;

    int gFailures = 0;
    int gChecks = 0;

    void Check( bool ok, const std::string& what )
    {
        gChecks++;
        if( !ok )
        {
            gFailures++;
            std::cout << "  FAIL: " << what << std::endl;
        }
    }

    struct FeatureSetName
    {
        FastSIMD::FeatureSet featureSet;
        const char* name;
    };

    constexpr FeatureSetName kFeatureSets[] = {
        { FastSIMD::FeatureSet::SSE2, "SSE2" },
        { FastSIMD::FeatureSet::SSE41, "SSE41" },
        { FastSIMD::FeatureSet::AVX2, "AVX2" },
        { FastSIMD::FeatureSet::AVX512, "AVX512" },
        { FastSIMD::FeatureSet::NEON, "NEON" },
        { FastSIMD::FeatureSet::AARCH64, "AARCH64" },
    };

    std::vector<FeatureSetName> AvailableFeatureSets()
    {
        // An explicit level is not checked against the CPU, only the default one is
        FastSIMD::FeatureSet best = FastNoise::New<FastNoise::Constant>()->GetActiveFeatureSet();

        std::vector<FeatureSetName> result;
        for( const FeatureSetName& entry: kFeatureSets )
        {
            auto node = entry.featureSet <= best ? FastNoise::New<FastNoise::Constant>( entry.featureSet ) : nullptr;
            if( node && node->GetActiveFeatureSet() == entry.featureSet )
            {
                result.push_back( entry );
            }
        }
        return result;
    }

    struct FormulaInfo
    {
        Formula formula;
        const char* name;
        std::function<SmartNode<FastNoise::VariableRange<FastNoise::Seeded<FastNoise::ScalableGenerator>>>( FastSIMD::FeatureSet )> create;
        float roundingSteps2D; // k: roundings from the input coordinate to the vertex offset, in 2D
        float roundingSteps3D;
        float roundingSteps4D;

        float RoundingSteps( int dims ) const
        {
            return dims == 2 ? roundingSteps2D : dims == 3 ? roundingSteps3D
                                                           : roundingSteps4D;
        }
    };

    template<typename T>
    SmartNode<FastNoise::VariableRange<FastNoise::Seeded<FastNoise::ScalableGenerator>>> CreateAs( FastSIMD::FeatureSet featureSet )
    {
        return FastNoise::New<T>( featureSet );
    }

    const FormulaInfo kFormulas[] = {
        // Scale, skew sum (1 add in 2D, 2 in 3D, 3 in 4D), skew multiply, skew add, fraction subtract, unskew sum (1 / 2 / 3), unskew multiply, unskew add
        { Formula::Simplex, "Simplex", CreateAs<FastNoise::Simplex>, 8, 10, 12 },
        // Scale, fraction subtract; floor is exact
        { Formula::Perlin, "Perlin", CreateAs<FastNoise::Perlin>, 2, 2, 2 },
        // Same chain as Simplex; the vertex moves are exact integer steps
        { Formula::SuperSimplex, "SuperSimplex", CreateAs<FastNoise::SuperSimplex>, 8, 10, 12 },
        // Scale, fraction subtract; floor is exact
        { Formula::Value, "Value", CreateAs<FastNoise::Value>, 2, 2, 2 },
    };

    std::unique_ptr<GradientProbe> NewProbe( FastSIMD::FeatureSet featureSet )
    {
        std::unique_ptr<GradientProbe> probe( FastSIMD::NewDispatchClass<GradientProbe>( featureSet ) );
        if( !probe || probe->GetActiveFeatureSet() != featureSet )
        {
            std::cout << "  probe not available for this feature set" << std::endl;
            gFailures++;
            return nullptr;
        }
        return probe;
    }

    int32_t UlpDistance( float a, float b )
    {
        int32_t ia, ib;
        std::memcpy( &ia, &a, sizeof( float ) );
        std::memcpy( &ib, &b, sizeof( float ) );
        if( ia < 0 )
            ia = INT32_MIN - ia;
        if( ib < 0 )
            ib = INT32_MIN - ib;
        int64_t d = (int64_t)ia - (int64_t)ib;
        return (int32_t)std::min<int64_t>( std::abs( d ), INT32_MAX );
    }

    // Grid points, a few on and next to cell edges, plus far points when requested
    std::vector<float> TestPoints( int dims, float spacing, bool far )
    {
        std::vector<float> points;
        auto add = [&]( float x, float y, float z, float w ) {
            points.push_back( x );
            points.push_back( y );
            if( dims >= 3 )
                points.push_back( z );
            if( dims == 4 )
                points.push_back( w );
        };

        for( int i = 0; i < 9; i++ )
        {
            for( int j = 0; j < 9; j++ )
            {
                add( ( i - 4 ) * spacing * 0.731f, ( j - 4 ) * spacing * 0.619f, ( i + j - 8 ) * spacing * 0.457f, ( i - j ) * spacing * 0.383f );
            }
        }
        for( float edge: { 0.0f, 1.0f, -3.0f, 17.0f } )
        {
            add( edge * spacing, 0.3f * spacing, 0.7f * spacing, 0.6f * spacing );
            add( ( edge + 1e-3f ) * spacing, ( edge - 1e-3f ) * spacing, 0.5f * spacing, edge * spacing );
            add( 0.2f * spacing, edge * spacing, ( edge + 1e-3f ) * spacing, ( edge - 1e-3f ) * spacing );
        }
        if( far )
        {
            for( float f: { 1.0e6f, -1.0e6f } )
            {
                add( f, f * 0.5f + 0.37f, -f + 0.71f, f * 0.75f + 0.2f );
                add( f + 0.125f, -f, f * 0.25f, -f * 0.5f );
            }
        }
        return points;
    }

    std::vector<float> GenPoints( const FastNoise::Generator* node, int dims, const std::vector<float>& points, int seed )
    {
        int count = (int)points.size() / dims;
        std::vector<float> xs( count ), ys( count ), zs( count ), ws( count ), out( count );
        for( int i = 0; i < count; i++ )
        {
            xs[i] = points[i * dims];
            ys[i] = points[i * dims + 1];
            zs[i] = dims >= 3 ? points[i * dims + 2] : 0.0f;
            ws[i] = dims == 4 ? points[i * dims + 3] : 0.0f;
        }
        if( dims == 2 )
        {
            node->GenPositionArray2D( out.data(), count, xs.data(), ys.data(), 0, 0, seed );
        }
        else if( dims == 3 )
        {
            node->GenPositionArray3D( out.data(), count, xs.data(), ys.data(), zs.data(), 0, 0, 0, seed );
        }
        else
        {
            node->GenPositionArray4D( out.data(), count, xs.data(), ys.data(), zs.data(), ws.data(), 0, 0, 0, 0, seed );
        }
        return out;
    }

    // In ulps of the output range's largest magnitude, see tests/damped/README.md
    // |fused - Gen| <= 4 * ulp( max( |Output Min|, |Output Max| ) )
    constexpr float kValueRangeUlpTol = 4.0f;

    std::vector<std::pair<std::string, float>> gWorstRangeUlpPerLevel;

    float RangeUlps( const std::vector<float>& a, const std::vector<float>& b, float rangeBound )
    {
        float rangeUlp = std::nextafter( rangeBound, 2 * rangeBound ) - rangeBound;
        float worst = 0;
        for( size_t i = 0; i < a.size(); i++ )
        {
            worst = std::max( worst, std::abs( a[i] - b[i] ) / rangeUlp );
        }
        return worst;
    }

    // |value - Gen| <= 4 range ulp + k * |grad| * ulp( largest lattice coordinate ), see tests/damped/README.md
    // k is 0 for GenGradient; gradients are per cell in output units
    void CheckValueBound( const std::vector<float>& values, const std::vector<float>& gen, const std::vector<float>& gradients, const std::vector<float>& positions,
                          const std::vector<float>& lattice, int dims, float rangeBound, float k, const char* level, const std::string& label )
    {
        float rangeUlp = std::nextafter( rangeBound, 2 * rangeBound ) - rangeBound;
        float worstRatio = -1, worstMeasured = 0, worstAllowed = 0, worstGrad = 0;
        size_t worstAt = 0;
        for( size_t i = 0; i < values.size(); i++ )
        {
            float gradLengthSq = 0, largest = 0;
            for( int d = 0; d < dims; d++ )
            {
                gradLengthSq += gradients[i * dims + d] * gradients[i * dims + d];
                largest = std::max( largest, std::abs( lattice[i * dims + d] ) );
            }
            float coordUlp = std::nextafter( largest, 2 * largest + 1 ) - largest;
            float allowed = kValueRangeUlpTol + k * std::sqrt( gradLengthSq ) * coordUlp / rangeUlp;
            float measured = std::abs( values[i] - gen[i] ) / rangeUlp;
            if( measured / allowed > worstRatio )
            {
                worstRatio = measured / allowed;
                worstMeasured = measured;
                worstAllowed = allowed;
                worstGrad = std::sqrt( gradLengthSq );
                worstAt = i;
            }
        }

        std::string position;
        for( int d = 0; d < dims; d++ )
        {
            position += ( d ? ", " : "" ) + std::to_string( positions[worstAt * dims + d] );
        }
        char detail[256];
        std::snprintf( detail, sizeof( detail ), "worst at (%s): value %.9g, Gen %.9g, |grad| %.4g, measured %.3g range ulp, allowed %.3g",
                       position.c_str(), values[worstAt], gen[worstAt], worstGrad, worstMeasured, worstAllowed );
        std::cout << "    " << label << ": " << detail << std::endl;
        Check( worstRatio <= 1.0f, label + " " + detail );

        if( k == 0 )
        {
            auto entry = std::find_if( gWorstRangeUlpPerLevel.begin(), gWorstRangeUlpPerLevel.end(), [&]( const auto& e ) { return e.first == level; } );
            if( entry == gWorstRangeUlpPerLevel.end() )
            {
                gWorstRangeUlpPerLevel.emplace_back( level, worstMeasured );
            }
            else
            {
                entry->second = std::max( entry->second, worstMeasured );
            }
        }
    }

    // Brief test 2, per feature set: values from the 3.1 functions and from GenGradient match Gen
    void TestValueMatchesGen( const FeatureSetName& fs )
    {
        std::cout << "[2] value vs Gen, " << fs.name << std::endl;
        auto probe = NewProbe( fs.featureSet );
        if( !probe )
            return;

        for( const FormulaInfo& info: kFormulas )
        {
            for( int dims: { 2, 3, 4 } )
            {
                std::string label = std::string( info.name ) + " " + std::to_string( dims ) + "D " + fs.name;

                // Static function on lattice positions scaled exactly as ScalePositions does, world points up to +-1e6
                {
                    auto node = info.create( fs.featureSet );
                    node->SetScale( 100.0f );
                    std::vector<float> points = TestPoints( dims, 100.0f, true );
                    std::vector<float> lattice = points;
                    for( float& p: lattice )
                    {
                        p *= 1.0f / 100.0f;
                    }
                    int count = (int)points.size() / dims;
                    std::vector<float> values( count ), gradients( count * dims );
                    probe->FormulaGradient( info.formula, dims, 1337, lattice.data(), count, values.data(), gradients.data(), true );
                    std::vector<float> gen = GenPoints( node.get(), dims, points, 1337 );

                    float bound = probe->FormulaBound( info.formula, dims );
                    for( float& g: gradients )
                    {
                        g /= bound;
                    }
                    float k = info.RoundingSteps( dims );
                    CheckValueBound( values, gen, gradients, points, lattice, dims, 1.0f, k, fs.name, label + " static function" );
                }

                // GenGradient override with non-default seed offset, Feature Scale and Output Min/Max
                {
                    auto node = info.create( fs.featureSet );
                    node->SetScale( 37.5f );
                    node->SetSeedOffset( 91 );
                    node->SetOutputMin( -3.0f );
                    node->SetOutputMax( 5.0f );
                    Check( probe->SetTarget( node ), "probe binds to its target" );
                    std::vector<float> points = TestPoints( dims, 37.5f, true );
                    int count = (int)points.size() / dims;
                    std::vector<float> values( count ), gradients( count * dims );
                    bool has = probe->NodeGradient( dims, -77, points.data(), count, values.data(), gradients.data() );
                    Check( has, label + " GenGradient reports a gradient" );
                    std::vector<float> gen = GenPoints( node.get(), dims, points, -77 );

                    std::vector<float> lattice = points;
                    for( float& p: lattice )
                    {
                        p *= 1.0f / 37.5f;
                    }
                    CheckValueBound( values, gen, gradients, points, lattice, dims, 5.0f, 0, fs.name, label + " GenGradient" );
                }
            }
        }
    }

    // Tolerance for analytic vs central-difference gradients, per noise cell
    constexpr float kGradientStep = 1.0e-3f; // central-difference step, in cells
    constexpr float kGradientAbsTol = 2.0e-2f; // absolute, per cell, in [-1, 1] output units
    constexpr float kGradientRelTol = 1.0e-2f; // relative to the gradient's length

    bool GradientClose( const float* analytic, const float* numeric, int dims, float& error )
    {
        float diff = 0, length = 0;
        for( int d = 0; d < dims; d++ )
        {
            diff += ( analytic[d] - numeric[d] ) * ( analytic[d] - numeric[d] );
            length += analytic[d] * analytic[d];
        }
        error = std::sqrt( diff );
        return error <= kGradientAbsTol + kGradientRelTol * std::sqrt( length );
    }

    // On a kink (3D Simplex boundaries) the analytic gradient matches one side, not the central difference
    bool GradientMatches( const float* analytic, const float* central, const float* forward, const float* backward, int dims, float& error )
    {
        if( GradientClose( analytic, central, dims, error ) )
        {
            return true;
        }
        float oneSided;
        if( GradientClose( analytic, forward, dims, oneSided ) || GradientClose( analytic, backward, dims, oneSided ) )
        {
            error = oneSided;
            return true;
        }
        return false;
    }

    // Brief test 3: analytic gradients match central differences of Gen, through both routes
    void TestGradientsMatchCentralDifferences( const FeatureSetName& fs )
    {
        std::cout << "[3] gradient vs central differences, " << fs.name << std::endl;
        auto probe = NewProbe( fs.featureSet );
        if( !probe )
            return;

        for( const FormulaInfo& info: kFormulas )
        {
            for( int dims: { 2, 3, 4 } )
            {
                std::string label = std::string( info.name ) + " " + std::to_string( dims ) + "D " + fs.name;

                // Static function on unit-lattice positions; native gradients divided by the bound are in [-1, 1] output units
                {
                    std::vector<float> points = TestPoints( dims, 1.0f, false );
                    int count = (int)points.size() / dims;
                    std::vector<float> values( count ), gradients( count * dims );
                    probe->FormulaGradient( info.formula, dims, 4321, points.data(), count, values.data(), gradients.data(), true );
                    float bound = probe->FormulaBound( info.formula, dims );

                    float worst = 0;
                    std::string worstDetail;
                    bool ok = true;
                    for( int i = 0; i < count; i++ )
                    {
                        float numeric[4], analytic[4], forward[4], backward[4];
                        for( int d = 0; d < dims; d++ )
                        {
                            float sample[4], unused[4], vPlus, vMinus;
                            std::memcpy( sample, &points[i * dims], dims * sizeof( float ) );
                            sample[d] += kGradientStep;
                            probe->FormulaGradient( info.formula, dims, 4321, sample, 1, &vPlus, unused, true );
                            sample[d] -= 2 * kGradientStep;
                            probe->FormulaGradient( info.formula, dims, 4321, sample, 1, &vMinus, unused, true );
                            numeric[d] = ( vPlus - vMinus ) / ( 2 * kGradientStep );
                            forward[d] = ( vPlus - values[i] ) / kGradientStep;
                            backward[d] = ( values[i] - vMinus ) / kGradientStep;
                            analytic[d] = gradients[i * dims + d] / bound;
                        }
                        float error;
                        ok &= GradientMatches( analytic, numeric, forward, backward, dims, error );
                        if( error > worst )
                        {
                            worst = error;
                            worstDetail = " at";
                            for( int d = 0; d < dims; d++ )
                                worstDetail += " " + std::to_string( points[i * dims + d] );
                            worstDetail += " analytic";
                            for( int d = 0; d < dims; d++ )
                                worstDetail += " " + std::to_string( analytic[d] );
                            worstDetail += " central";
                            for( int d = 0; d < dims; d++ )
                                worstDetail += " " + std::to_string( numeric[d] );
                            worstDetail += " forward";
                            for( int d = 0; d < dims; d++ )
                                worstDetail += " " + std::to_string( forward[d] );
                        }
                    }
                    Check( ok, label + " static function vs central differences, worst error " + std::to_string( worst ) + worstDetail );
                }

                // GenGradient override: world-space central differences times Feature Scale give per-cell slopes
                {
                    constexpr float kScale = 37.5f;
                    auto node = info.create( fs.featureSet );
                    node->SetScale( kScale );
                    node->SetSeedOffset( 91 );
                    node->SetOutputMin( -3.0f );
                    node->SetOutputMax( 5.0f );
                    Check( probe->SetTarget( node ), "probe binds to its target" );

                    std::vector<float> points = TestPoints( dims, kScale, false );
                    int count = (int)points.size() / dims;
                    std::vector<float> values( count ), gradients( count * dims );
                    probe->NodeGradient( dims, -77, points.data(), count, values.data(), gradients.data() );

                    float worst = 0;
                    bool ok = true;
                    for( int i = 0; i < count; i++ )
                    {
                        // Output range is 8 wide, so divide by 4 to compare in [-1, 1] units
                        float numeric[4], forward[4], backward[4], analytic[4];
                        for( int d = 0; d < dims; d++ )
                        {
                            std::vector<float> plus( points.begin() + i * dims, points.begin() + ( i + 1 ) * dims );
                            std::vector<float> minus = plus;
                            plus[d] += kGradientStep * kScale;
                            minus[d] -= kGradientStep * kScale;
                            float vPlus = GenPoints( node.get(), dims, plus, -77 )[0];
                            float vMinus = GenPoints( node.get(), dims, minus, -77 )[0];
                            numeric[d] = ( vPlus - vMinus ) / ( 2 * kGradientStep ) / 4.0f;
                            forward[d] = ( vPlus - values[i] ) / kGradientStep / 4.0f;
                            backward[d] = ( values[i] - vMinus ) / kGradientStep / 4.0f;
                            analytic[d] = gradients[i * dims + d] / 4.0f;
                        }
                        float error;
                        ok &= GradientMatches( analytic, numeric, forward, backward, dims, error );
                        worst = std::max( worst, error );
                    }
                    Check( ok, label + " GenGradient vs central differences, worst error " + std::to_string( worst ) );
                }
            }
        }
    }
} // namespace

namespace
{
    using Damped = FastNoise::FractalDamped;

    SmartNode<Damped> NewDamped( FastNoise::SmartNodeArg<> source, int octaves = 3 )
    {
        auto node = FastNoise::New<Damped>();
        node->SetSource( source );
        node->SetOctaveCount( octaves );
        return node;
    }

    float MaxAbsDiff( const std::vector<float>& a, const std::vector<float>& b )
    {
        float worst = 0;
        for( size_t i = 0; i < a.size(); i++ )
            worst = std::max( worst, std::abs( a[i] - b[i] ) );
        return worst;
    }

    std::vector<float> Scaled( std::vector<float> points, float factor )
    {
        for( float& p: points )
            p *= factor;
        return points;
    }

    // Brief test 4: Feature Scale does not shrink the gradient, Output Min/Max scales it
    void TestUnits()
    {
        std::cout << "[4] gradient units" << std::endl;
        for( int dims: { 2, 3, 4 } )
        {
            auto small = FastNoise::New<FastNoise::Simplex>();
            auto large = FastNoise::New<FastNoise::Simplex>();
            small->SetScale( 100 );
            large->SetScale( 2000 );

            std::vector<float> points = TestPoints( dims, 100.0f, false );
            float diff = MaxAbsDiff( GenPoints( NewDamped( small ).get(), dims, points, 3 ), GenPoints( NewDamped( large ).get(), dims, Scaled( points, 20 ), 3 ) );
            Check( diff <= 1e-4f, std::to_string( dims ) + "D Feature Scale 100 vs 2000 at 20x positions, max diff " + std::to_string( diff ) );
        }

        auto probe = NewProbe( FastNoise::New<FastNoise::Constant>()->GetActiveFeatureSet() );
        if( !probe )
            return;
        for( int dims: { 2, 3, 4 } )
        {
            auto unit = FastNoise::New<FastNoise::Simplex>();
            auto ranged = FastNoise::New<FastNoise::Simplex>();
            ranged->SetOutputMin( -3 );
            ranged->SetOutputMax( 5 );

            std::vector<float> points = TestPoints( dims, 100.0f, false );
            int count = (int)points.size() / dims;
            std::vector<float> values( count ), gUnit( count * dims ), gRanged( count * dims );
            Check( probe->SetTarget( unit ), "probe binds to its target" );
            probe->NodeGradient( dims, 9, points.data(), count, values.data(), gUnit.data() );
            Check( probe->SetTarget( ranged ), "probe binds to its target" );
            probe->NodeGradient( dims, 9, points.data(), count, values.data(), gRanged.data() );

            float worst = 0;
            for( size_t i = 0; i < gUnit.size(); i++ )
                worst = std::max( worst, std::abs( gRanged[i] - 4 * gUnit[i] ) / ( 1e-3f + std::abs( 4 * gUnit[i] ) ) );
            Check( worst <= 1e-5f, std::to_string( dims ) + "D Output Min/Max -3..5 scales the gradient by 4, worst relative error " + std::to_string( worst ) );
        }
    }

    // Domain, bounds and margins are explained in tests/damped/README.md
    struct ApproxMode
    {
        const char* name;
        Damped::Mode mode;
        float maxTol;
        float meanTol2D;
        float meanTol;
        float stepsPerCell; // the mode's eps is cell / this
    };
    const ApproxMode kApproxModes[] = {
        { "ApproximateFast", Damped::Mode::ApproximateFast, 0.3f, 0.05f, 0.03f, 256 },
        { "ApproximateAccurate", Damped::Mode::ApproximateAccurate, 0.3f, 0.03f, 0.03f, 64 },
    };

    struct ApproxResult
    {
        std::string level, region;
        int dims;
        float maxDiff[2], meanDiff[2];
    };
    std::vector<ApproxResult> gApproxResults;

    // Near the origin (|pos| up to about 1900, 19 cells at Feature Scale 100), or around a far centre given in cells
    std::vector<float> ApproxPoints( int dims, float centreCells )
    {
        std::vector<float> points = TestPoints( dims, 1000.0f / 9, false );
        for( float& p: points )
        {
            p += centreCells * 100.0f;
        }
        return points;
    }

    // Brief test 5: Analytic and both Approximate modes agree on a Simplex source, near the origin and at 1e5 and 1e6 cells
    void TestAnalyticMatchesApproximate( const FeatureSetName& fs )
    {
        std::cout << "[5] Analytic vs Approximate, " << fs.name << std::endl;
        auto source = FastNoise::New<FastNoise::Simplex>( fs.featureSet );

        for( int dims: { 2, 3, 4 } )
        {
            for( float centre: { 0.0f, 1.0e5f, 1.0e6f } )
            {
                ApproxResult result { fs.name, centre == 0 ? "near" : centre == 1.0e5f ? "1e5 cells"
                                                                                       : "1e6 cells",
                                      dims,
                                      {},
                                      {} };
                std::vector<float> points = ApproxPoints( dims, centre );

                auto analytic = FastNoise::New<Damped>( fs.featureSet );
                analytic->SetSource( source );
                std::vector<float> reference = GenPoints( analytic.get(), dims, points, 11 );

                for( int m = 0; m < 2; m++ )
                {
                    auto approximate = FastNoise::New<Damped>( fs.featureSet );
                    approximate->SetSource( source );
                    approximate->SetMode( kApproxModes[m].mode );
                    std::vector<float> values = GenPoints( approximate.get(), dims, points, 11 );

                    float maxDiff = 0, mean = 0;
                    for( size_t i = 0; i < values.size(); i++ )
                    {
                        float diff = std::abs( values[i] - reference[i] );
                        maxDiff = std::max( maxDiff, std::isfinite( diff ) ? diff : INFINITY );
                        mean += diff / values.size();
                    }
                    result.maxDiff[m] = maxDiff;
                    result.meanDiff[m] = mean;
                    std::cout << "    " << kApproxModes[m].name << " " << dims << "D " << result.region << ": max " << maxDiff << ", mean " << mean << std::endl;

                    std::string label = std::string( kApproxModes[m].name ) + " " + std::to_string( dims ) + "D " + result.region + " " + fs.name;
                    if( centre > 1.0e5f )
                    {
                        // Outside the domain: reported only, positions resolve to 0.08-0.3 cells here
                        Check( std::isfinite( maxDiff ) && std::isfinite( mean ), label + " finite" );
                        continue;
                    }
                    float meanTol = dims == 2 ? kApproxModes[m].meanTol2D : kApproxModes[m].meanTol;
                    Check( maxDiff <= kApproxModes[m].maxTol && mean <= meanTol,
                           label + " within max " + std::to_string( kApproxModes[m].maxTol ) + ", mean " + std::to_string( meanTol ) );
                }
                gApproxResults.push_back( result );
            }
        }
    }

    // One octave of each Approximate mode against differences built by hand with the same step rule
    void TestApproximateByHand( const FeatureSetName& fs )
    {
        std::cout << "[5] Approximate octaves built by hand, " << fs.name << std::endl;
        auto source = FastNoise::New<FastNoise::Simplex>( fs.featureSet );
        constexpr float kCell = 100.0f;

        // At 1e5 cells p + eps rounds (realised step 2 vs eps 1.5625), so a compiler folding (p + eps) - p back to eps shows here
        for( int dims: { 2, 3, 4 } )
            for( float centre: { 0.0f, 1.0e5f } )
            {
                std::vector<float> points = ApproxPoints( dims, centre );
                int count = (int)points.size() / dims;
                std::vector<float> n = GenPoints( source.get(), dims, points, 11 );

                for( int m = 0; m < 2; m++ )
                {
                    std::vector<float> expected( count );
                    for( int i = 0; i < count; i++ )
                    {
                        float lengthSq = 0;
                        for( int d = 0; d < dims; d++ )
                        {
                            std::vector<float> plus( points.begin() + i * dims, points.begin() + ( i + 1 ) * dims ), minus = plus;
                            float p = plus[d];
                            float step = std::max( kCell / kApproxModes[m].stepsPerCell, std::abs( p ) / ( 1 << 21 ) );
                            plus[d] = p + step;
                            minus[d] = p - step;
                            float fPlus = GenPoints( source.get(), dims, plus, 11 )[0];
                            float g = m == 0 ? ( fPlus - n[i] ) / ( plus[d] - p ) * kCell
                                             : ( fPlus - GenPoints( source.get(), dims, minus, 11 )[0] ) / ( plus[d] - minus[d] ) * kCell;
                            lengthSq += g * g;
                        }
                        expected[i] = n[i] / ( 1 + lengthSq );
                    }

                    auto oneOctave = FastNoise::New<Damped>( fs.featureSet );
                    oneOctave->SetSource( source );
                    oneOctave->SetOctaveCount( 1 );
                    oneOctave->SetMode( kApproxModes[m].mode );
                    float handDiff = MaxAbsDiff( GenPoints( oneOctave.get(), dims, points, 11 ), expected );
                    std::cout << "    " << kApproxModes[m].name << " " << dims << "D at " << centre << " cells, one octave by hand: max diff " << handDiff << std::endl;
                    Check( handDiff <= 1e-5f, std::string( kApproxModes[m].name ) + " " + std::to_string( dims ) + "D at " + std::to_string( centre ) + " cells, one octave by hand, max diff " + std::to_string( handDiff ) );
                }
            }
    }

    // ApproximateAccurate earns its place only by a lower maximum error than ApproximateFast in every measured case
    void ReportApproximateComparison()
    {
        std::cout << "Analytic vs Approximate, max / mean (level, dims, region: Fast | Accurate)" << std::endl;
        for( const ApproxResult& r: gApproxResults )
        {
            std::cout << "    " << r.level << " " << r.dims << "D " << r.region << ": " << r.maxDiff[0] << " / " << r.meanDiff[0] << " | "
                      << r.maxDiff[1] << " / " << r.meanDiff[1] << std::endl;
            bool inDomain = r.region != "1e6 cells";
            if( inDomain && r.dims <= 3 )
            {
                Check( r.maxDiff[1] < r.maxDiff[0], std::string( "ApproximateAccurate beats ApproximateFast on max error, " ) + r.level + " " + std::to_string( r.dims ) + "D " + r.region );
            }
        }
    }

    // Brief test 6: Auto equals the explicit formula when the slope settings match the source's own
    void TestAutoMatchesExplicit()
    {
        std::cout << "[6] Auto vs explicit Perlin" << std::endl;
        for( int dims: { 2, 3, 4 } )
        {
            auto source = FastNoise::New<FastNoise::Perlin>();
            source->SetScale( 64 );
            source->SetSeedOffset( 5 );
            auto autoNode = NewDamped( source, 4 );
            auto explicitNode = NewDamped( source, 4 );
            explicitNode->SetSlopeFormula( Damped::SlopeFormula::Perlin );
            explicitNode->SetSlopeFeatureScale( 64 );
            explicitNode->SetSlopeSeedOffset( 5 );

            // Output bound is the sum of the octave amplitudes, 1 + 0.5 + 0.25 + 0.125
            std::vector<float> points = TestPoints( dims, 64.0f, true );
            float worst = RangeUlps( GenPoints( autoNode.get(), dims, points, 21 ), GenPoints( explicitNode.get(), dims, points, 21 ), 1.875f );
            Check( worst <= kValueRangeUlpTol, std::to_string( dims ) + "D Auto equals explicit Perlin within " + std::to_string( (int)kValueRangeUlpTol ) + " range ulp, measured " + std::to_string( worst ) );
        }
    }

    // Brief test 7: a zero gradient gives FractalFBm, a real one does not
    void TestZeroGradientIsFBm()
    {
        std::cout << "[7] zero gradient equals FractalFBm" << std::endl;
        for( int dims: { 2, 3, 4 } )
        {
            std::vector<float> points = TestPoints( dims, 50.0f, false );

            auto constant = FastNoise::New<FastNoise::Constant>();
            constant->SetValue( 0.625f );
            auto damped = NewDamped( constant, 5 );
            auto fbm = FastNoise::New<FastNoise::FractalFBm>();
            fbm->SetSource( constant );
            fbm->SetOctaveCount( 5 );
            fbm->SetGain( 0.45f );
            damped->SetGain( 0.45f );
            fbm->SetWeightedStrength( 0.3f );
            damped->SetWeightedStrength( 0.3f );
            for( auto mode: { Damped::Mode::Analytic, Damped::Mode::ApproximateFast, Damped::Mode::ApproximateAccurate } )
            {
                damped->SetMode( mode );
                std::vector<float> a = GenPoints( damped.get(), dims, points, 5 ), b = GenPoints( fbm.get(), dims, points, 5 );
                float diff = MaxAbsDiff( a, b );
                Check( diff == 0.0f, std::to_string( dims ) + "D Constant source equals FractalFBm, mode " + std::to_string( (int)mode ) + ", max diff " + std::to_string( diff ) );
            }

            auto simplex = FastNoise::New<FastNoise::Simplex>();
            auto realDamped = NewDamped( simplex, 5 );
            auto realFbm = FastNoise::New<FastNoise::FractalFBm>();
            realFbm->SetSource( simplex );
            realFbm->SetOctaveCount( 5 );
            float diff = MaxAbsDiff( GenPoints( realDamped.get(), dims, points, 5 ), GenPoints( realFbm.get(), dims, points, 5 ) );
            Check( diff > 1e-3f, std::to_string( dims ) + "D Simplex source differs from FractalFBm, max diff " + std::to_string( diff ) );
        }
    }

    // Brief test 8: each octave's value and slope use the same seed and position, checked against octaves built by hand
    void TestOctaveInputs()
    {
        std::cout << "[8] octaves built by hand" << std::endl;
        auto probe = NewProbe( FastNoise::New<FastNoise::Constant>()->GetActiveFeatureSet() );
        if( !probe )
            return;

        constexpr float kRotation = 36.87f;
        constexpr float kLacunarity = 2.0f;
        constexpr float kGain = 0.5f;
        float c = std::cos( kRotation * ( 3.14159265358979323846f / 180.0f ) );
        float s = std::sin( kRotation * ( 3.14159265358979323846f / 180.0f ) );

        for( int dims: { 2, 3, 4 } )
        {
            auto source = FastNoise::New<FastNoise::Simplex>();
            source->SetSeedOffset( 3 );
            Check( probe->SetTarget( source ), "probe binds to its target" );

            std::vector<float> points = TestPoints( dims, 80.0f, false );
            int count = (int)points.size() / dims;
            constexpr int kSeed = 1234;

            std::vector<float> n0( count ), g0( count * dims ), n1( count ), g1( count * dims );
            probe->NodeGradient( dims, kSeed, points.data(), count, n0.data(), g0.data() );

            std::vector<float> next( points.size() );
            for( int i = 0; i < count; i++ )
            {
                const float* p = &points[i * dims];
                float* q = &next[i * dims];
                q[0] = ( c * p[0] - s * p[1] ) * kLacunarity;
                q[1] = ( s * p[0] + c * p[1] ) * kLacunarity;
                for( int d = 2; d < dims; d++ )
                    q[d] = p[d] * kLacunarity;
            }
            probe->NodeGradient( dims, kSeed + 1, next.data(), count, n1.data(), g1.data() );

            std::vector<float> one( count ), two( count );
            for( int i = 0; i < count; i++ )
            {
                float d0 = 0, d01 = 0;
                for( int d = 0; d < dims; d++ )
                {
                    d0 += g0[i * dims + d] * g0[i * dims + d];
                    float sum = g0[i * dims + d] + g1[i * dims + d];
                    d01 += sum * sum;
                }
                one[i] = n0[i] / ( 1 + d0 );
                two[i] = one[i] + kGain * n1[i] / ( 1 + d01 );
            }

            float diffOne = MaxAbsDiff( GenPoints( NewDamped( source, 1 ).get(), dims, points, kSeed ), one );
            float diffTwo = MaxAbsDiff( GenPoints( NewDamped( source, 2 ).get(), dims, points, kSeed ), two );
            Check( diffOne <= 1e-6f, std::to_string( dims ) + "D one octave by hand, max diff " + std::to_string( diffOne ) );
            Check( diffTwo <= 1e-5f, std::to_string( dims ) + "D two octaves by hand, max diff " + std::to_string( diffTwo ) );
        }
    }

    // Brief test 9: registration, members and encoding
    void TestMetadata()
    {
        std::cout << "[9] metadata" << std::endl;
        const auto& all = FastNoise::Metadata::GetAll();
        Check( all.size() == 48, "48 nodes registered, found " + std::to_string( all.size() ) );
        const FastNoise::Metadata& meta = FastNoise::Metadata::Get<Damped>();
        Check( meta.id == 47 && all[all.size() - 1] == &meta, "FractalDamped is the last ID" );
        Check( std::strcmp( meta.name, "FractalDamped" ) == 0, "class name" );
        Check( FastNoise::Metadata::FormatMetadataNodeName( &meta ) == "Fractal Damped", "display name " + FastNoise::Metadata::FormatMetadataNodeName( &meta ) );
        Check( meta.groups.size() == 1 && std::strcmp( meta.groups[0], "Fractal" ) == 0, "group Fractal" );
        Check( std::strlen( meta.description ) > 0, "description" );

        Check( meta.memberNodeLookups.size() == 1 && std::strcmp( meta.memberNodeLookups[0].name, "Source" ) == 0, "Source lookup" );
        Check( meta.memberHybrids.size() == 2 &&
                   std::strcmp( meta.memberHybrids[0].name, "Gain" ) == 0 && meta.memberHybrids[0].valueDefault == 0.5f &&
                   std::strcmp( meta.memberHybrids[1].name, "Weighted Strength" ) == 0 && meta.memberHybrids[1].valueDefault == 0.0f,
               "inherited hybrids" );

        using Var = FastNoise::Metadata::MemberVariable;
        struct Expected
        {
            const char* name;
            Var::eType type;
            Var::ValueUnion def;
        };
        const Expected expected[] = {
            { "Octaves", Var::EInt, 3 },
            { "Lacunarity", Var::EFloat, 2.0f },
            { "Rotation", Var::EFloat, 36.87f },
            { "Mode", Var::EEnum, 0 },
            { "Slope Formula", Var::EEnum, 0 },
            { "Slope Feature Scale", Var::EFloat, 100.0f },
            { "Slope Seed Offset", Var::EInt, 0 },
        };
        Check( meta.memberVariables.size() == std::size( expected ), "member variable count" );
        for( size_t i = 0; i < std::min( meta.memberVariables.size(), std::size( expected ) ); i++ )
        {
            const Var& v = meta.memberVariables[i];
            Check( std::strcmp( v.name, expected[i].name ) == 0 && v.type == expected[i].type && v.valueDefault == expected[i].def,
                   std::string( "member " ) + expected[i].name );
            Check( std::strlen( v.description ) > 0, std::string( "member description " ) + expected[i].name );
        }
        for( const auto& h: meta.memberHybrids )
            Check( std::strlen( h.description ) > 0, std::string( "hybrid description " ) + h.name );
        Check( meta.memberVariables[0].valueMin.i == 2 && meta.memberVariables[0].valueMax.i == 16, "Octaves range inherited" );

        auto enumIs = []( const Var& v, std::initializer_list<const char*> names ) {
            if( v.enumNames.size() != names.size() )
                return false;
            size_t i = 0;
            for( const char* n: names )
                if( std::strcmp( v.enumNames[i++], n ) != 0 )
                    return false;
            return true;
        };
        Check( enumIs( meta.memberVariables[3], { "Analytic", "ApproximateFast", "ApproximateAccurate" } ), "Mode strings" );
        Check( enumIs( meta.memberVariables[4], { "Auto", "Simplex", "Perlin", "SuperSimplex", "Value" } ), "Slope Formula strings" );

        // Encode and decode a tree with every member off its default
        std::vector<std::unique_ptr<FastNoise::NodeData>> store;
        store.emplace_back( std::make_unique<FastNoise::NodeData>( &meta ) );
        FastNoise::NodeData* root = store.back().get();
        store.emplace_back( std::make_unique<FastNoise::NodeData>( &FastNoise::Metadata::Get<FastNoise::Perlin>() ) );
        root->nodeLookups[0] = store.back().get();
        store.emplace_back( std::make_unique<FastNoise::NodeData>( &FastNoise::Metadata::Get<FastNoise::Value>() ) );
        root->hybrids[0].first = store.back().get();
        root->hybrids[1].second = 0.25f;
        root->variables[0] = 5;
        root->variables[1] = 2.5f;
        root->variables[2] = 12.0f;
        root->variables[3] = 1;
        root->variables[4] = 2;
        root->variables[5] = 80.0f;
        root->variables[6] = 7;

        std::string encoded = FastNoise::Metadata::SerialiseNodeData( root );
        std::vector<std::unique_ptr<FastNoise::NodeData>> decodedStore;
        FastNoise::NodeData* decoded = FastNoise::Metadata::DeserialiseNodeData( encoded.c_str(), decodedStore );
        Check( decoded && decoded->metadata == &meta && decoded->variables == root->variables &&
                   decoded->hybrids[1].second == 0.25f && decoded->hybrids[0].first && decoded->nodeLookups[0],
               "encode/decode round-trip" );
        Check( decoded && FastNoise::Metadata::SerialiseNodeData( decoded ) == encoded, "re-encoding gives the same string" );

        auto fromTree = FastNoise::NewFromEncodedNodeTree( encoded.c_str() );
        auto byHand = FastNoise::New<Damped>();
        byHand->SetSource( FastNoise::New<FastNoise::Perlin>() );
        byHand->SetGain( FastNoise::New<FastNoise::Value>() );
        byHand->SetWeightedStrength( 0.25f );
        byHand->SetOctaveCount( 5 );
        byHand->SetLacunarity( 2.5f );
        byHand->SetRotation( 12.0f );
        byHand->SetMode( Damped::Mode::ApproximateFast );
        byHand->SetSlopeFormula( Damped::SlopeFormula::Perlin );
        byHand->SetSlopeFeatureScale( 80.0f );
        byHand->SetSlopeSeedOffset( 7 );
        std::vector<float> points = TestPoints( 3, 40.0f, false );
        Check( fromTree && MaxAbsDiff( GenPoints( fromTree.get(), 3, points, 1 ), GenPoints( byHand.get(), 3, points, 1 ) ) == 0.0f, "decoded tree generates like the hand-built node" );

        for( int i = 0; i < 5; i++ )
        {
            auto state = FastNoise::Metadata::DebugCheckVectorStorageSize( i );
            Check( state.first < 0, std::string( "gMetadataVectorSize<" ) + state.second + "> should be " + std::to_string( std::abs( state.first ) ) );
        }
    }
} // namespace

int main()
{
    std::vector<FeatureSetName> featureSets = AvailableFeatureSets();
    std::cout << "Feature sets:";
    for( auto& fs: featureSets )
        std::cout << " " << fs.name;
    std::cout << std::endl;

    for( auto& fs: featureSets )
    {
        TestValueMatchesGen( fs );
        TestGradientsMatchCentralDifferences( fs );
        TestAnalyticMatchesApproximate( fs );
        TestApproximateByHand( fs );
    }

    TestUnits();
    ReportApproximateComparison();
    TestAutoMatchesExplicit();
    TestZeroGradientIsFBm();
    TestOctaveInputs();
    TestMetadata();

    std::cout << "Largest GenGradient value vs Gen difference per SIMD level, in range ulps:";
    for( const auto& entry: gWorstRangeUlpPerLevel )
    {
        std::cout << " " << entry.first << " " << entry.second;
    }
    std::cout << std::endl;

    std::cout << gChecks << " checks, " << gFailures << " failures" << std::endl;
    return gFailures ? 1 : 0;
}
