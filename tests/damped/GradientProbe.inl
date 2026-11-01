#include "GradientProbe.h"

// Generator.inl defines the SIMD types the others use
#include <FastNoise/Generators/Generator.inl>

#include <FastNoise/Generators/BasicGenerators.inl>
#include <FastNoise/Generators/Perlin.inl>
#include <FastNoise/Generators/Simplex.inl>
#include <FastNoise/Generators/Value.inl>

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<GradientProbe, SIMD> final : public virtual GradientProbe, public FastSIMD::DispatchClass<Generator, SIMD>
{
    static constexpr int kLanes = (int)float32v::ElementCount;

    float32v FS_VECTORCALL Gen( int32v, float32v, float32v ) const override
    {
        return float32v( 0 );
    }
    float32v FS_VECTORCALL Gen( int32v, float32v, float32v, float32v ) const override
    {
        return float32v( 0 );
    }

    // Loads lanes of one axis, repeating the last point past the end
    static float32v LoadAxis( const float* positions, int dims, int axis, int start, int count )
    {
        alignas( 64 ) float lanes[kLanes];
        for( int l = 0; l < kLanes; l++ )
        {
            int point = start + l < count ? start + l : count - 1;
            lanes[l] = positions[point * dims + axis];
        }
        return FS::Load<float32v>( lanes );
    }

    static FS_FORCEINLINE void StoreLanes( float32v v, float* out, int stride, int offset, int start, int count )
    {
        alignas( 64 ) float lanes[kLanes];
        FS::Store( lanes, v );
        for( int l = 0; l < kLanes && start + l < count; l++ )
        {
            out[( start + l ) * stride + offset] = lanes[l];
        }
    }

    bool NodeGradient( int dims, int seed, const float* positions, int count, float* values, float* gradients ) const override
    {
        if( !mTarget.simdGeneratorPtr )
        {
            return false;
        }
        auto target = this->GetSourceSIMD( mTarget );

        for( int start = 0; start < count; start += kLanes )
        {
            float32v value, g[3];
            bool has;
            if( dims == 2 )
            {
                has = target->GenGradient( int32v( seed ),
                                           LoadAxis( positions, 2, 0, start, count ), LoadAxis( positions, 2, 1, start, count ),
                                           value, g[0], g[1] );
            }
            else
            {
                has = target->GenGradient( int32v( seed ),
                                           LoadAxis( positions, 3, 0, start, count ), LoadAxis( positions, 3, 1, start, count ), LoadAxis( positions, 3, 2, start, count ),
                                           value, g[0], g[1], g[2] );
            }
            if( !has )
            {
                return false;
            }

            StoreLanes( value, values, 1, 0, start, count );
            for( int d = 0; d < dims; d++ )
            {
                StoreLanes( g[d], gradients, dims, d, start, count );
            }
        }
        return true;
    }

    template<typename T>
    static FS_FORCEINLINE float32v Formula2D( int32v seed, float32v x, float32v y, float32v& gx, float32v& gy )
    {
        return FastSIMD::DispatchClass<T, SIMD>::ValueGradient( seed, x, y, gx, gy );
    }

    template<typename T>
    static FS_FORCEINLINE float32v Formula3D( int32v seed, float32v x, float32v y, float32v z, float32v& gx, float32v& gy, float32v& gz )
    {
        return FastSIMD::DispatchClass<T, SIMD>::ValueGradient( seed, x, y, z, gx, gy, gz );
    }

    template<typename T>
    static float NativeBound( int dims )
    {
        if constexpr( std::is_same_v<T, FastNoise::Value> )
        {
            return kValueBounds;
        }
        else
        {
            return dims == 2 ? FastSIMD::DispatchClass<T, SIMD>::kBounding2D : FastSIMD::DispatchClass<T, SIMD>::kBounding3D;
        }
    }

    template<typename T>
    void RunFormula( int dims, int seed, const float* positions, int count, float* values, float* gradients, bool scaleOutput ) const
    {
        float bound = NativeBound<T>( dims );

        for( int start = 0; start < count; start += kLanes )
        {
            float32v value, g[3];
            if( dims == 2 )
            {
                value = Formula2D<T>( int32v( seed ),
                                      LoadAxis( positions, 2, 0, start, count ), LoadAxis( positions, 2, 1, start, count ), g[0], g[1] );
            }
            else
            {
                value = Formula3D<T>( int32v( seed ),
                                      LoadAxis( positions, 3, 0, start, count ), LoadAxis( positions, 3, 1, start, count ), LoadAxis( positions, 3, 2, start, count ),
                                      g[0], g[1], g[2] );
            }

            if( scaleOutput )
            {
                // Same operations as VariableRange's ScaleOutput with Output Min -1 and Output Max 1
                value = FS::FMulAdd( float32v( 1.0f / ( bound - -bound ) ) * float32v( 2.0f ), value - float32v( -bound ), float32v( -1.0f ) );
            }

            StoreLanes( value, values, 1, 0, start, count );
            for( int d = 0; d < dims; d++ )
            {
                StoreLanes( g[d], gradients, dims, d, start, count );
            }
        }
    }

    void FormulaGradient( Formula formula, int dims, int seed, const float* positions, int count, float* values, float* gradients, bool scaleOutput ) const override
    {
        switch( formula )
        {
        case Formula::Simplex:
            RunFormula<FastNoise::Simplex>( dims, seed, positions, count, values, gradients, scaleOutput );
            break;
        case Formula::Perlin:
            RunFormula<FastNoise::Perlin>( dims, seed, positions, count, values, gradients, scaleOutput );
            break;
        case Formula::SuperSimplex:
            RunFormula<FastNoise::SuperSimplex>( dims, seed, positions, count, values, gradients, scaleOutput );
            break;
        case Formula::Value:
            RunFormula<FastNoise::Value>( dims, seed, positions, count, values, gradients, scaleOutput );
            break;
        }
    }

    float FormulaBound( Formula formula, int dims ) const override
    {
        switch( formula )
        {
        case Formula::Simplex:
            return NativeBound<FastNoise::Simplex>( dims );
        case Formula::Perlin:
            return NativeBound<FastNoise::Perlin>( dims );
        case Formula::SuperSimplex:
            return NativeBound<FastNoise::SuperSimplex>( dims );
        case Formula::Value:
            return NativeBound<FastNoise::Value>( dims );
        }
        return 0;
    }
};

template class FastSIMD::RegisterDispatchClass<GradientProbe>;
