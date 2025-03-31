; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 663376

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child336 child307 - child
    bread430 bread235 - bread-portion
    content243 content191 - content-portion
    tray266 tray444 tray136 tray162 - tray
    table268 table281 table141 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray266 kitchen)
     (at tray444 kitchen)
     (at tray136 kitchen)
     (at tray162 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread235)
     (at_kitchen_content content243)
     (at_kitchen_content content191)
     (not_allergic_gluten child336)
     (not_allergic_gluten child307)
     (waiting child336 table141)
     (waiting child307 table141)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child336)
     (served child307)
    )
  )
)
