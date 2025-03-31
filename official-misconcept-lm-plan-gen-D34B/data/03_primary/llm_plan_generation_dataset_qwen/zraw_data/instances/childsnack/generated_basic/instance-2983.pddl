; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 914479

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 child96 - child
    bread254 bread25 - bread-portion
    content405 content220 - content-portion
    tray363 tray266 tray352 tray346 - tray
    table112 table116 table357 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray266 kitchen)
     (at tray352 kitchen)
     (at tray346 kitchen)
     (at_kitchen_bread bread254)
     (at_kitchen_bread bread25)
     (at_kitchen_content content405)
     (at_kitchen_content content220)
     (not_allergic_gluten child198)
     (not_allergic_gluten child96)
     (waiting child198 table357)
     (waiting child96 table112)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child198)
     (served child96)
    )
  )
)
