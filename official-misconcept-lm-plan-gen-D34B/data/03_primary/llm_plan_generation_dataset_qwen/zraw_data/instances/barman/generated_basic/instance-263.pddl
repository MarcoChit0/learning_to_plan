(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot221 shot82 shot315 - shot
      ingredient74 ingredient124 ingredient422 - ingredient
      cocktail1 - cocktail
      dispenser299 dispenser158 dispenser110 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot221)
  (ontable shot82)
  (ontable shot315)
  (dispenses dispenser299 ingredient74)
  (dispenses dispenser158 ingredient124)
  (dispenses dispenser110 ingredient422)
  (clean shaker316)
  (clean shot221)
  (clean shot82)
  (clean shot315)
  (empty shaker316)
  (empty shot221)
  (empty shot82)
  (empty shot315)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient124)
  (cocktail-part2 cocktail1 ingredient422)
)
 (:goal
  (and
      (contains shot221 cocktail1)
      (contains shot82 cocktail1)
)))
