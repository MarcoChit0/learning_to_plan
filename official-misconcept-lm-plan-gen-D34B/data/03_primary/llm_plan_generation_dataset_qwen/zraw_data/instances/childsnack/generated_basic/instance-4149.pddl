; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 373764

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child88 child179 - child
    bread320 bread230 - bread-portion
    content200 content141 - content-portion
    tray304 tray190 tray42 - tray
    table107 table3 table216 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray304 kitchen)
     (at tray190 kitchen)
     (at tray42 kitchen)
     (at_kitchen_bread bread320)
     (at_kitchen_bread bread230)
     (at_kitchen_content content200)
     (at_kitchen_content content141)
     (not_allergic_gluten child88)
     (not_allergic_gluten child179)
     (waiting child88 table216)
     (waiting child179 table3)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child88)
     (served child179)
    )
  )
)
