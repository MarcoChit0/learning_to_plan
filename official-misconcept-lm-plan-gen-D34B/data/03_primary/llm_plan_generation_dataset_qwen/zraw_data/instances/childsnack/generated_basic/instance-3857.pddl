; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 259941

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 - child
    bread187 - bread-portion
    content138 - content-portion
    tray107 tray338 tray368 tray84 - tray
    table216 table224 table296 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray107 kitchen)
     (at tray338 kitchen)
     (at tray368 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread187)
     (at_kitchen_content content138)
     (not_allergic_gluten child431)
     (waiting child431 table216)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child431)
    )
  )
)
