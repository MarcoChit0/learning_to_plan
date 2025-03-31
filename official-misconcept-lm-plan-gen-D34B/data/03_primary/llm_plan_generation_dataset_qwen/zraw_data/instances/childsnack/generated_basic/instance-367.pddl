; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 751678

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 - child
    bread66 - bread-portion
    content337 - content-portion
    tray120 tray135 tray187 tray122 - tray
    table492 table96 table223 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray120 kitchen)
     (at tray135 kitchen)
     (at tray187 kitchen)
     (at tray122 kitchen)
     (at_kitchen_bread bread66)
     (at_kitchen_content content337)
     (not_allergic_gluten child151)
     (waiting child151 table223)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child151)
    )
  )
)
