(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot270 shot38 - shot
      ingredient129 ingredient105 ingredient31 ingredient78 - ingredient
      cocktail231 - cocktail
      dispenser444 dispenser479 dispenser425 dispenser155 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot270)
  (ontable shot38)
  (dispenses dispenser444 ingredient129)
  (dispenses dispenser479 ingredient105)
  (dispenses dispenser425 ingredient31)
  (dispenses dispenser155 ingredient78)
  (clean shaker25)
  (clean shot270)
  (clean shot38)
  (empty shaker25)
  (empty shot270)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail231 ingredient129)
  (cocktail-part2 cocktail231 ingredient31)
)
 (:goal
  (and
      (contains shot270 cocktail231)
)))
