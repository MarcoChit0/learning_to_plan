(define (problem prob)
 (:domain barman)
 (:objects 
      shaker448 - shaker
      left right - hand
      shot110 shot323 shot429 - shot
      ingredient25 ingredient131 ingredient249 - ingredient
      cocktail459 - cocktail
      dispenser126 dispenser65 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker448)
  (ontable shot110)
  (ontable shot323)
  (ontable shot429)
  (dispenses dispenser126 ingredient25)
  (dispenses dispenser65 ingredient131)
  (dispenses dispenser363 ingredient249)
  (clean shaker448)
  (clean shot110)
  (clean shot323)
  (clean shot429)
  (empty shaker448)
  (empty shot110)
  (empty shot323)
  (empty shot429)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker448 l0)
  (shaker-level shaker448 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient131)
  (cocktail-part2 cocktail459 ingredient25)
)
 (:goal
  (and
      (contains shot110 cocktail459)
      (contains shot323 ingredient25)
)))
