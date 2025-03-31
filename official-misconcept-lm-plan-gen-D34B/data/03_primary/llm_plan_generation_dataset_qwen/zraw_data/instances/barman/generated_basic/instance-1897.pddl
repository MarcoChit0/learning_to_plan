(define (problem prob)
 (:domain barman)
 (:objects 
      shaker35 - shaker
      left right - hand
      shot214 shot490 shot20 shot49 - shot
      ingredient365 ingredient88 - ingredient
      cocktail177 - cocktail
      dispenser317 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker35)
  (ontable shot214)
  (ontable shot490)
  (ontable shot20)
  (ontable shot49)
  (dispenses dispenser317 ingredient365)
  (dispenses dispenser441 ingredient88)
  (clean shaker35)
  (clean shot214)
  (clean shot490)
  (clean shot20)
  (clean shot49)
  (empty shaker35)
  (empty shot214)
  (empty shot490)
  (empty shot20)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker35 l0)
  (shaker-level shaker35 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail177 ingredient88)
  (cocktail-part2 cocktail177 ingredient365)
)
 (:goal
  (and
      (contains shot214 cocktail177)
      (contains shot490 cocktail177)
      (contains shot20 ingredient365)
)))
