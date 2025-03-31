(define (problem prob)
 (:domain barman)
 (:objects 
      shaker279 - shaker
      left right - hand
      shot332 shot456 - shot
      ingredient473 ingredient355 - ingredient
      cocktail47 - cocktail
      dispenser275 dispenser55 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker279)
  (ontable shot332)
  (ontable shot456)
  (dispenses dispenser275 ingredient473)
  (dispenses dispenser55 ingredient355)
  (clean shaker279)
  (clean shot332)
  (clean shot456)
  (empty shaker279)
  (empty shot332)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker279 l0)
  (shaker-level shaker279 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail47 ingredient355)
  (cocktail-part2 cocktail47 ingredient473)
)
 (:goal
  (and
      (contains shot332 cocktail47)
)))
