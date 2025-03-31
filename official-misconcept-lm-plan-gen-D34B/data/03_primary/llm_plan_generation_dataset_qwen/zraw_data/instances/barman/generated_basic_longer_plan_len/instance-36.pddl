(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot143 shot317 shot326 - shot
      ingredient101 ingredient68 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser221 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot143)
  (ontable shot317)
  (ontable shot326)
  (dispenses dispenser221 ingredient101)
  (dispenses dispenser6 ingredient68)
  (clean shaker180)
  (clean shot143)
  (clean shot317)
  (clean shot326)
  (empty shaker180)
  (empty shot143)
  (empty shot317)
  (empty shot326)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient68)
  (cocktail-part2 cocktail1 ingredient101)
  (cocktail-part1 cocktail2 ingredient101)
  (cocktail-part2 cocktail2 ingredient68)
)
 (:goal
  (and
      (contains shot143 cocktail2)
      (contains shot317 cocktail1)
)))
