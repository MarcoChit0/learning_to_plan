(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot77 shot466 - shot
      ingredient83 ingredient134 ingredient377 - ingredient
      cocktail1 - cocktail
      dispenser110 dispenser70 dispenser88 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot77)
  (ontable shot466)
  (dispenses dispenser110 ingredient83)
  (dispenses dispenser70 ingredient134)
  (dispenses dispenser88 ingredient377)
  (clean shaker288)
  (clean shot77)
  (clean shot466)
  (empty shaker288)
  (empty shot77)
  (empty shot466)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient83)
  (cocktail-part2 cocktail1 ingredient377)
)
 (:goal
  (and
      (contains shot77 cocktail1)
)))
