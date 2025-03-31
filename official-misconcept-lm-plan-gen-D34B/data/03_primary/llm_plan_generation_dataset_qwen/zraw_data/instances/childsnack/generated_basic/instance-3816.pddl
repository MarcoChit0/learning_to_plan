; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 277623

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 - child
    bread478 - bread-portion
    content484 - content-portion
    tray317 tray67 tray245 tray278 - tray
    table212 table70 table468 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray317 kitchen)
     (at tray67 kitchen)
     (at tray245 kitchen)
     (at tray278 kitchen)
     (at_kitchen_bread bread478)
     (at_kitchen_content content484)
     (not_allergic_gluten child441)
     (waiting child441 table70)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child441)
    )
  )
)
