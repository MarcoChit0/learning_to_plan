(define (problem prob)
 (:domain barman)
 (:objects 
      shaker420 - shaker
      left right - hand
      shot419 shot76 - shot
      ingredient25 ingredient95 ingredient69 ingredient67 - ingredient
      cocktail128 - cocktail
      dispenser155 dispenser236 dispenser256 dispenser169 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker420)
  (ontable shot419)
  (ontable shot76)
  (dispenses dispenser155 ingredient25)
  (dispenses dispenser236 ingredient95)
  (dispenses dispenser256 ingredient69)
  (dispenses dispenser169 ingredient67)
  (clean shaker420)
  (clean shot419)
  (clean shot76)
  (empty shaker420)
  (empty shot419)
  (empty shot76)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker420 l0)
  (shaker-level shaker420 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail128 ingredient69)
  (cocktail-part2 cocktail128 ingredient95)
)
 (:goal
  (and
      (contains shot419 cocktail128)
)))
