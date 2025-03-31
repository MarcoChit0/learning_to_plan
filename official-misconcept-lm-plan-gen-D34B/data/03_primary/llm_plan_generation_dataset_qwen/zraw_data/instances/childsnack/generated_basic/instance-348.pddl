; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 579501

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 - child
    bread221 - bread-portion
    content403 - content-portion
    tray256 tray363 tray336 tray215 - tray
    table482 table91 table253 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray256 kitchen)
     (at tray363 kitchen)
     (at tray336 kitchen)
     (at tray215 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_content content403)
     (not_allergic_gluten child462)
     (waiting child462 table91)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child462)
    )
  )
)
