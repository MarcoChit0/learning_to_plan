(define (problem prob)
 (:domain barman)
 (:objects 
      shaker47 - shaker
      left right - hand
      shot360 shot319 shot63 - shot
      ingredient268 ingredient353 ingredient446 ingredient371 - ingredient
      cocktail129 - cocktail
      dispenser177 dispenser287 dispenser209 dispenser349 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker47)
  (ontable shot360)
  (ontable shot319)
  (ontable shot63)
  (dispenses dispenser177 ingredient268)
  (dispenses dispenser287 ingredient353)
  (dispenses dispenser209 ingredient446)
  (dispenses dispenser349 ingredient371)
  (clean shaker47)
  (clean shot360)
  (clean shot319)
  (clean shot63)
  (empty shaker47)
  (empty shot360)
  (empty shot319)
  (empty shot63)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker47 l0)
  (shaker-level shaker47 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail129 ingredient353)
  (cocktail-part2 cocktail129 ingredient268)
)
 (:goal
  (and
      (contains shot360 cocktail129)
      (contains shot319 cocktail129)
)))
