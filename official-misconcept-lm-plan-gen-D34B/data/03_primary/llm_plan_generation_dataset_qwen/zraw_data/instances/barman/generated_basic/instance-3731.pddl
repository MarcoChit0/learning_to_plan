(define (problem prob)
 (:domain barman)
 (:objects 
      shaker340 - shaker
      left right - hand
      shot447 shot499 shot303 - shot
      ingredient491 ingredient496 ingredient355 - ingredient
      cocktail274 - cocktail
      dispenser188 dispenser490 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker340)
  (ontable shot447)
  (ontable shot499)
  (ontable shot303)
  (dispenses dispenser188 ingredient491)
  (dispenses dispenser490 ingredient496)
  (dispenses dispenser196 ingredient355)
  (clean shaker340)
  (clean shot447)
  (clean shot499)
  (clean shot303)
  (empty shaker340)
  (empty shot447)
  (empty shot499)
  (empty shot303)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker340 l0)
  (shaker-level shaker340 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient355)
  (cocktail-part2 cocktail274 ingredient491)
)
 (:goal
  (and
      (contains shot447 cocktail274)
      (contains shot499 cocktail274)
)))
