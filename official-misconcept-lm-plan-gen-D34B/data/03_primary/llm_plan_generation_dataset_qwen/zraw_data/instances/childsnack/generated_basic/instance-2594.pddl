; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 711123

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child30 - child
    bread438 - bread-portion
    content223 - content-portion
    tray398 tray283 tray111 - tray
    table191 table457 table151 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at tray283 kitchen)
     (at tray111 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_content content223)
     (not_allergic_gluten child30)
     (waiting child30 table191)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child30)
    )
  )
)
