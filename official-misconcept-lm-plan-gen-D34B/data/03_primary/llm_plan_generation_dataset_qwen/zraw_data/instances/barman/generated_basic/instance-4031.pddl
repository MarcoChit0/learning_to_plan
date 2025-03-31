(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot322 shot267 - shot
      ingredient185 ingredient412 ingredient221 - ingredient
      cocktail76 - cocktail
      dispenser192 dispenser63 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot322)
  (ontable shot267)
  (dispenses dispenser192 ingredient185)
  (dispenses dispenser63 ingredient412)
  (dispenses dispenser221 ingredient221)
  (clean shaker157)
  (clean shot322)
  (clean shot267)
  (empty shaker157)
  (empty shot322)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail76 ingredient412)
  (cocktail-part2 cocktail76 ingredient221)
)
 (:goal
  (and
      (contains shot322 cocktail76)
)))
