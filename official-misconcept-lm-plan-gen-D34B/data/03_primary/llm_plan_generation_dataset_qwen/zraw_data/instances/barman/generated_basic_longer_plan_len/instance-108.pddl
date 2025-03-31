(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot324 shot309 shot494 - shot
      ingredient367 ingredient84 ingredient48 ingredient386 - ingredient
      cocktail1 - cocktail
      dispenser456 dispenser455 dispenser5 dispenser129 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot324)
  (ontable shot309)
  (ontable shot494)
  (dispenses dispenser456 ingredient367)
  (dispenses dispenser455 ingredient84)
  (dispenses dispenser5 ingredient48)
  (dispenses dispenser129 ingredient386)
  (clean shaker207)
  (clean shot324)
  (clean shot309)
  (clean shot494)
  (empty shaker207)
  (empty shot324)
  (empty shot309)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient84)
  (cocktail-part2 cocktail1 ingredient386)
)
 (:goal
  (and
      (contains shot324 cocktail1)
      (contains shot309 ingredient367)
)))
