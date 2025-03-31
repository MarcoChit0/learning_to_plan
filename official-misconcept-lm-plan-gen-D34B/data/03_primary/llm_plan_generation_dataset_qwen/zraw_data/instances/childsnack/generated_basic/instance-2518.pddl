; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 137733

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child236 - child
    bread45 - bread-portion
    content369 - content-portion
    tray283 tray344 tray158 - tray
    table440 table238 table166 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray283 kitchen)
     (at tray344 kitchen)
     (at tray158 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_content content369)
     (not_allergic_gluten child236)
     (waiting child236 table238)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child236)
    )
  )
)
