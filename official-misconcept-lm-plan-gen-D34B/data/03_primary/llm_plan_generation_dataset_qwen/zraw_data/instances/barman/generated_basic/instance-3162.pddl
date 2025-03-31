(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot396 shot84 shot116 - shot
      ingredient216 ingredient168 ingredient23 ingredient163 - ingredient
      cocktail165 - cocktail
      dispenser245 dispenser193 dispenser173 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot396)
  (ontable shot84)
  (ontable shot116)
  (dispenses dispenser245 ingredient216)
  (dispenses dispenser193 ingredient168)
  (dispenses dispenser173 ingredient23)
  (dispenses dispenser44 ingredient163)
  (clean shaker383)
  (clean shot396)
  (clean shot84)
  (clean shot116)
  (empty shaker383)
  (empty shot396)
  (empty shot84)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient216)
  (cocktail-part2 cocktail165 ingredient163)
)
 (:goal
  (and
      (contains shot396 cocktail165)
      (contains shot84 cocktail165)
)))
