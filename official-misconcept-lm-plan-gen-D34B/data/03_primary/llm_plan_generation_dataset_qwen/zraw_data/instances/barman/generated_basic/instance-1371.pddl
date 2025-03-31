(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot392 - shot
      ingredient314 ingredient79 ingredient448 - ingredient
      cocktail94 - cocktail
      dispenser151 dispenser200 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot392)
  (dispenses dispenser151 ingredient314)
  (dispenses dispenser200 ingredient79)
  (dispenses dispenser221 ingredient448)
  (clean shaker29)
  (clean shot392)
  (empty shaker29)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail94 ingredient314)
  (cocktail-part2 cocktail94 ingredient448)
)
 (:goal
  (and
      (contains shot392 cocktail94)
)))
