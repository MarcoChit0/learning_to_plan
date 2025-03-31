; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 14214

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child150 child141 - child
    bread41 bread156 - bread-portion
    content56 content403 - content-portion
    tray105 tray414 tray253 - tray
    table395 table154 table129 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at tray414 kitchen)
     (at tray253 kitchen)
     (at_kitchen_bread bread41)
     (at_kitchen_bread bread156)
     (at_kitchen_content content56)
     (at_kitchen_content content403)
     (no_gluten_bread bread156)
     (no_gluten_content content56)
     (allergic_gluten child150)
     (not_allergic_gluten child141)
     (waiting child150 table129)
     (waiting child141 table395)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child150)
     (served child141)
    )
  )
)
