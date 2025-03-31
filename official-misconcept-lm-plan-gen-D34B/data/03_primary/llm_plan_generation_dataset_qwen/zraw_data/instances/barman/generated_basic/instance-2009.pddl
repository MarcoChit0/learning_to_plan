(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot162 shot137 shot456 - shot
      ingredient304 ingredient275 ingredient181 - ingredient
      cocktail339 - cocktail
      dispenser361 dispenser484 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot162)
  (ontable shot137)
  (ontable shot456)
  (dispenses dispenser361 ingredient304)
  (dispenses dispenser484 ingredient275)
  (dispenses dispenser477 ingredient181)
  (clean shaker478)
  (clean shot162)
  (clean shot137)
  (clean shot456)
  (empty shaker478)
  (empty shot162)
  (empty shot137)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail339 ingredient275)
  (cocktail-part2 cocktail339 ingredient181)
)
 (:goal
  (and
      (contains shot162 cocktail339)
      (contains shot137 cocktail339)
)))
