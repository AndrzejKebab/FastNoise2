#include "Fractal.h"

template<FastSIMD::FeatureSet SIMD, typename T>
class FastSIMD::DispatchClass<Fractal<T>, SIMD> : public virtual Fractal<T>, public DispatchClass<Generator, SIMD>
{

};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FractalFBm, SIMD> final : public virtual FractalFBm, public DispatchClass<Fractal<>, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v gain = this->GetSourceValue( mGain  , seed, pos... );
        float32v weightedStrength = this->GetSourceValue( mWeightedStrength, seed, pos... );
        float32v lacunarity( mLacunarity );
        float32v amp( 1.0f );
        float32v noise = this->GetSourceValue( mSource, seed, pos... );

        float32v sum = noise * amp;

        for( int i = 1; i < mOctaves; i++ )
        {
            seed -= int32v( -1 );
            amp *= Lerp( float32v( 1 ), (noise + float32v( 1 )) * float32v( 0.5f ), weightedStrength );
            amp *= gain;

            noise = this->GetSourceValue( mSource, seed, (pos *= lacunarity)... );
            sum += noise * amp;
        }

        return sum;
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FractalRidged, SIMD> final : public virtual FractalRidged, public DispatchClass<Fractal<>, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT(int32v seed, P... pos) const
    {
        float32v gain = this->GetSourceValue( mGain, seed, pos... );
        float32v weightedStrength = this->GetSourceValue( mWeightedStrength, seed, pos... );
        float32v lacunarity( mLacunarity );
        float32v amp( 1.0f );
        float32v noise = FS::Abs( this->GetSourceValue( mSource, seed, pos... ) );

        float32v sum = (noise * float32v( -2 ) + float32v( 1 )) * amp;

        for( int i = 1; i < mOctaves; i++ )
        {
            seed -= int32v( -1 );
            amp *= Lerp( float32v( 1 ), float32v( 1 ) - noise, weightedStrength );
            amp *= gain;

            noise = FS::Abs( this->GetSourceValue( mSource, seed, (pos *= lacunarity)... ) );
            sum += (noise * float32v( -2 ) + float32v( 1 )) * amp;
        }

        return sum;
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<FractalDamped, SIMD> final : public virtual FractalDamped, public DispatchClass<Fractal<>, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        constexpr size_t kDims = sizeof...( P );

        float32v gain = this->GetSourceValue( mGain, seed, pos... );
        float32v weightedStrength = this->GetSourceValue( mWeightedStrength, seed, pos... );
        float32v lacunarity( mLacunarity );
        float32v rotationCos( mRotationCos );
        float32v rotationSin( mRotationSin );
        float32v amp( 1.0f );
        int32v slopeSeed = seed + int32v( mSlopeSeedOffset );

        float32v position[kDims] = { pos... };
        float32v slope[kDims];
        float32v noise = OctaveValueGradient( seed, slopeSeed, position, slope, std::make_index_sequence<kDims> {} );

        // Same arithmetic as FractalFBm, so a zero slope gives its exact output
        float32v sum = noise * Damp( amp, slope );

        for( int i = 1; i < mOctaves; i++ )
        {
            seed -= int32v( -1 );
            slopeSeed -= int32v( -1 );
            amp *= Lerp( float32v( 1 ), ( noise + float32v( 1 ) ) * float32v( 0.5f ), weightedStrength );
            amp *= gain;

            float32v x = position[0];
            position[0] = FS::FNMulAdd( rotationSin, position[1], rotationCos * x );
            position[1] = FS::FMulAdd( rotationSin, x, rotationCos * position[1] );
            for( size_t d = 0; d < kDims; d++ )
            {
                position[d] *= lacunarity;
            }

            float32v gradient[kDims];
            noise = OctaveValueGradient( seed, slopeSeed, position, gradient, std::make_index_sequence<kDims> {} );

            // Unweighted, in this octave's own frame
            for( size_t d = 0; d < kDims; d++ )
            {
                slope[d] += gradient[d];
            }

            sum = FS::FMulAdd( noise, Damp( amp, slope ), sum );
        }

        return sum;
    }

    // amp / (1 + d.d); /fp:fast does not keep amp / 1 exact, so a zero slope returns amp untouched
    template<size_t N>
    static FS_FORCEINLINE float32v Damp( float32v amp, const float32v ( &slope )[N] )
    {
        float32v lengthSq = slope[0] * slope[0];
        for( size_t d = 1; d < N; d++ )
        {
            lengthSq = FS::FMulAdd( slope[d], slope[d], lengthSq );
        }
        return FS::Select( lengthSq > float32v( 0 ), amp / ( float32v( 1 ) + lengthSq ), amp );
    }

    template<size_t... I>
    FS_FORCEINLINE float32v OctaveValueGradient( int32v seed, int32v slopeSeed, const float32v ( &position )[sizeof...( I )], float32v ( &gradient )[sizeof...( I )], std::index_sequence<I...> ) const
    {
        constexpr size_t kDims = sizeof...( I );

        if( mMode == Mode::Analytic )
        {
            if( mSlopeFormula == SlopeFormula::Auto )
            {
                float32v value;
                if( this->GetSourceSIMD( mSource )->GenGradient( seed, position[I]..., value, gradient[I]... ) )
                {
                    return value;
                }
            }
            else
            {
                float32v slopeFrequency( 1.0f / mSlopeFeatureScale );
                float32v slopePosition[kDims] = { ( position[I] * slopeFrequency )... };

                using SimplexSIMD = FastSIMD::DispatchClass<Simplex, SIMD>;
                using PerlinSIMD = FastSIMD::DispatchClass<Perlin, SIMD>;
                using SuperSimplexSIMD = FastSIMD::DispatchClass<SuperSimplex, SIMD>;

                switch( mSlopeFormula )
                {
                case SlopeFormula::Simplex:
                    SimplexSIMD::ValueGradient( slopeSeed, slopePosition[I]..., gradient[I]... );
                    NormaliseGradient( gradient, NativeBound<SimplexSIMD>( kDims ) );
                    break;
                case SlopeFormula::Perlin:
                    PerlinSIMD::ValueGradient( slopeSeed, slopePosition[I]..., gradient[I]... );
                    NormaliseGradient( gradient, NativeBound<PerlinSIMD>( kDims ) );
                    break;
                case SlopeFormula::SuperSimplex:
                    SuperSimplexSIMD::ValueGradient( slopeSeed, slopePosition[I]..., gradient[I]... );
                    NormaliseGradient( gradient, NativeBound<SuperSimplexSIMD>( kDims ) );
                    break;
                case SlopeFormula::Value:
                    FastSIMD::DispatchClass<Value, SIMD>::ValueGradient( slopeSeed, slopePosition[I]..., gradient[I]... );
                    NormaliseGradient( gradient, kValueBounds );
                    break;
                default:
                    break;
                }

                return this->GetSourceValue( mSource, seed, position[I]... );
            }
        }

        // Finite differences with eps = cell / 64, divided by the step the positions actually moved
        float cellSize = this->GetSourceSIMD( mSource )->SlopeCellSize();
        float32v cell( cellSize != 0 ? cellSize : mSlopeFeatureScale );
        float32v epsilon = cell * float32v( 1.0f / 64 );

        float32v value = this->GetSourceValue( mSource, seed, position[I]... );
        if( mMode == Mode::ApproximateAccurate )
        {
            ( ( gradient[I] = CentralDifference<I>( seed, position, epsilon, cell ) ), ... );
        }
        else
        {
            ( ( gradient[I] = ForwardDifference<I>( seed, position, epsilon, cell, value ) ), ... );
        }
        return value;
    }

    // Far from the origin eps can be below the position's float spacing; the floor keeps the step a few ulps wide
    static FS_FORCEINLINE float32v StepSize( float32v position, float32v epsilon )
    {
        return FS::Max( epsilon, FS::Abs( position ) * float32v( 1.0f / ( 1 << 21 ) ) );
    }

    // Fast-math folds ( p + eps ) - p back to eps; XOR with a zero the compiler cannot see keeps the rounded position
    static FS_FORCEINLINE float32v Opaque( float32v value )
    {
        static volatile int32_t sZero = 0;
        return FS::Cast<float>( FS::Cast<int32_t>( value ) ^ int32v( sZero ) );
    }

    template<size_t AXIS, size_t N>
    FS_FORCEINLINE float32v ForwardDifference( int32v seed, const float32v ( &position )[N], float32v epsilon, float32v cell, float32v value ) const
    {
        float32v moved[N];
        std::copy( position, position + N, moved );
        moved[AXIS] = Opaque( position[AXIS] + StepSize( position[AXIS], epsilon ) );

        float32v realisedStep = moved[AXIS] - position[AXIS];
        return ( SampleAt( seed, moved, std::make_index_sequence<N> {} ) - value ) / realisedStep * cell;
    }

    template<size_t AXIS, size_t N>
    FS_FORCEINLINE float32v CentralDifference( int32v seed, const float32v ( &position )[N], float32v epsilon, float32v cell ) const
    {
        float32v step = StepSize( position[AXIS], epsilon );
        float32v plus[N], minus[N];
        std::copy( position, position + N, plus );
        std::copy( position, position + N, minus );
        plus[AXIS] = Opaque( position[AXIS] + step );
        minus[AXIS] = Opaque( position[AXIS] - step );

        float32v realisedStep = plus[AXIS] - minus[AXIS];
        return ( SampleAt( seed, plus, std::make_index_sequence<N> {} ) - SampleAt( seed, minus, std::make_index_sequence<N> {} ) ) / realisedStep * cell;
    }

    template<size_t N, size_t... I>
    FS_FORCEINLINE float32v SampleAt( int32v seed, const float32v ( &position )[N], std::index_sequence<I...> ) const
    {
        return this->GetSourceValue( mSource, seed, position[I]... );
    }

    template<typename T>
    static constexpr float NativeBound( size_t dims )
    {
        return dims == 2 ? T::kBounding2D : dims == 3 ? T::kBounding3D
                                                      : T::kBounding4D;
    }

    // Gradient of the formula's [-1, 1] normalised output
    template<size_t N>
    static FS_FORCEINLINE void NormaliseGradient( float32v ( &gradient )[N], float nativeBound )
    {
        float32v scale( 1.0f / nativeBound );
        for( size_t d = 0; d < N; d++ )
        {
            gradient[d] *= scale;
        }
    }
};
