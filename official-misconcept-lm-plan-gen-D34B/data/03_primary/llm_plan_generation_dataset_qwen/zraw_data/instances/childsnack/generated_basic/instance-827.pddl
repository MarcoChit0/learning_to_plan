; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 8113

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 - child
    bread321 - bread-portion
    content403 - content-portion
    tray86 tray225 - tray
    table247 table310 table355 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray86 kitchen)
     (at tray225 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_content content403)
     (not_allergic_gluten child362)
     (waiting child362 table247)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child362)
    )
  )
)
