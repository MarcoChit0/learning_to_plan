(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot257 shot372 - shot
      ingredient84 ingredient164 ingredient43 ingredient56 - ingredient
      cocktail228 - cocktail
      dispenser405 dispenser56 dispenser6 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot257)
  (ontable shot372)
  (dispenses dispenser405 ingredient84)
  (dispenses dispenser56 ingredient164)
  (dispenses dispenser6 ingredient43)
  (dispenses dispenser461 ingredient56)
  (clean shaker477)
  (clean shot257)
  (clean shot372)
  (empty shaker477)
  (empty shot257)
  (empty shot372)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient43)
  (cocktail-part2 cocktail228 ingredient164)
)
 (:goal
  (and
      (contains shot257 cocktail228)
)))
