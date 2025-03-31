; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 866585

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child178 child243 child407 - child
    bread265 bread448 bread458 - bread-portion
    content104 content43 content261 - content-portion
    tray191 tray490 tray359 - tray
    table94 table493 table19 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray490 kitchen)
     (at tray359 kitchen)
     (at_kitchen_bread bread265)
     (at_kitchen_bread bread448)
     (at_kitchen_bread bread458)
     (at_kitchen_content content104)
     (at_kitchen_content content43)
     (at_kitchen_content content261)
     (not_allergic_gluten child178)
     (not_allergic_gluten child407)
     (not_allergic_gluten child243)
     (waiting child178 table94)
     (waiting child243 table94)
     (waiting child407 table493)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child178)
     (served child243)
     (served child407)
    )
  )
)
