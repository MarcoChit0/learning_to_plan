; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 355515

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child235 - child
    bread267 - bread-portion
    content56 - content-portion
    tray167 tray380 tray94 tray75 - tray
    table440 table282 table176 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray380 kitchen)
     (at tray94 kitchen)
     (at tray75 kitchen)
     (at_kitchen_bread bread267)
     (at_kitchen_content content56)
     (not_allergic_gluten child235)
     (waiting child235 table440)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child235)
    )
  )
)
