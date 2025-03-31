(define (problem prob)
 (:domain barman)
 (:objects 
      shaker323 - shaker
      left right - hand
      shot4 shot146 shot183 - shot
      ingredient221 ingredient191 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser51 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker323)
  (ontable shot4)
  (ontable shot146)
  (ontable shot183)
  (dispenses dispenser51 ingredient221)
  (dispenses dispenser391 ingredient191)
  (clean shaker323)
  (clean shot4)
  (clean shot146)
  (clean shot183)
  (empty shaker323)
  (empty shot4)
  (empty shot146)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker323 l0)
  (shaker-level shaker323 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient191)
  (cocktail-part2 cocktail1 ingredient221)
  (cocktail-part1 cocktail2 ingredient221)
  (cocktail-part2 cocktail2 ingredient191)
)
 (:goal
  (and
      (contains shot4 cocktail2)
      (contains shot146 cocktail1)
)))
