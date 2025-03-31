(define (problem prob)
 (:domain barman)
 (:objects 
      shaker144 - shaker
      left right - hand
      shot361 shot0 - shot
      ingredient348 ingredient149 - ingredient
      cocktail61 - cocktail
      dispenser459 dispenser182 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker144)
  (ontable shot361)
  (ontable shot0)
  (dispenses dispenser459 ingredient348)
  (dispenses dispenser182 ingredient149)
  (clean shaker144)
  (clean shot361)
  (clean shot0)
  (empty shaker144)
  (empty shot361)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker144 l0)
  (shaker-level shaker144 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail61 ingredient348)
  (cocktail-part2 cocktail61 ingredient149)
)
 (:goal
  (and
      (contains shot361 cocktail61)
)))
