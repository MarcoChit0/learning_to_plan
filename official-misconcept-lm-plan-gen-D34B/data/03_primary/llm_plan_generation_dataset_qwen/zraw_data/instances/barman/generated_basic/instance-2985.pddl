(define (problem prob)
 (:domain barman)
 (:objects 
      shaker302 - shaker
      left right - hand
      shot46 shot199 - shot
      ingredient158 ingredient184 ingredient449 ingredient10 - ingredient
      cocktail386 - cocktail
      dispenser92 dispenser271 dispenser409 dispenser301 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker302)
  (ontable shot46)
  (ontable shot199)
  (dispenses dispenser92 ingredient158)
  (dispenses dispenser271 ingredient184)
  (dispenses dispenser409 ingredient449)
  (dispenses dispenser301 ingredient10)
  (clean shaker302)
  (clean shot46)
  (clean shot199)
  (empty shaker302)
  (empty shot46)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker302 l0)
  (shaker-level shaker302 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail386 ingredient449)
  (cocktail-part2 cocktail386 ingredient184)
)
 (:goal
  (and
      (contains shot46 cocktail386)
)))
