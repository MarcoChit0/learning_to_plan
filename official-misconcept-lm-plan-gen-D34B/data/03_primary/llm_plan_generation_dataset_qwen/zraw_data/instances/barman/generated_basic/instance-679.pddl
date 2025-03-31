(define (problem prob)
 (:domain barman)
 (:objects 
      shaker165 - shaker
      left right - hand
      shot287 shot158 - shot
      ingredient88 ingredient169 - ingredient
      cocktail1 - cocktail
      dispenser391 dispenser231 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker165)
  (ontable shot287)
  (ontable shot158)
  (dispenses dispenser391 ingredient88)
  (dispenses dispenser231 ingredient169)
  (clean shaker165)
  (clean shot287)
  (clean shot158)
  (empty shaker165)
  (empty shot287)
  (empty shot158)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker165 l0)
  (shaker-level shaker165 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient169)
  (cocktail-part2 cocktail1 ingredient88)
)
 (:goal
  (and
      (contains shot287 cocktail1)
)))
