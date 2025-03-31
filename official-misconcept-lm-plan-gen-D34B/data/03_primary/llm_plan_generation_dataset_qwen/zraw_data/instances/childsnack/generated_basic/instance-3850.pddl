; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 38113

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 - child
    bread365 - bread-portion
    content108 - content-portion
    tray296 tray62 tray226 tray224 - tray
    table226 table114 table102 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray296 kitchen)
     (at tray62 kitchen)
     (at tray226 kitchen)
     (at tray224 kitchen)
     (at_kitchen_bread bread365)
     (at_kitchen_content content108)
     (not_allergic_gluten child31)
     (waiting child31 table226)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child31)
    )
  )
)
