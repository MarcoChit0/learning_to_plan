(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot6 shot90 - shot
      ingredient426 ingredient88 ingredient358 ingredient79 - ingredient
      cocktail1 - cocktail
      dispenser241 dispenser339 dispenser291 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot6)
  (ontable shot90)
  (dispenses dispenser241 ingredient426)
  (dispenses dispenser339 ingredient88)
  (dispenses dispenser291 ingredient358)
  (dispenses dispenser126 ingredient79)
  (clean shaker425)
  (clean shot6)
  (clean shot90)
  (empty shaker425)
  (empty shot6)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient88)
  (cocktail-part2 cocktail1 ingredient79)
)
 (:goal
  (and
      (contains shot6 cocktail1)
)))
