; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 11933

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child16 - child
    bread411 - bread-portion
    content97 - content-portion
    tray82 tray258 tray295 tray358 - tray
    table85 table472 table251 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray82 kitchen)
     (at tray258 kitchen)
     (at tray295 kitchen)
     (at tray358 kitchen)
     (at_kitchen_bread bread411)
     (at_kitchen_content content97)
     (not_allergic_gluten child16)
     (waiting child16 table472)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child16)
    )
  )
)
