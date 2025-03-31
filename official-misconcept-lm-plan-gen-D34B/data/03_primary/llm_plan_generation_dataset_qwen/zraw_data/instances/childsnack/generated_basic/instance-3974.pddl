; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 668756

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child168 - child
    bread36 - bread-portion
    content152 - content-portion
    tray325 tray421 tray260 - tray
    table21 table89 table258 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray325 kitchen)
     (at tray421 kitchen)
     (at tray260 kitchen)
     (at_kitchen_bread bread36)
     (at_kitchen_content content152)
     (not_allergic_gluten child168)
     (waiting child168 table89)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child168)
    )
  )
)
