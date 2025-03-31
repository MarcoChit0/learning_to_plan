; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 874804

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child221 - child
    bread37 - bread-portion
    content78 - content-portion
    tray313 tray33 tray45 - tray
    table58 table427 table205 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray313 kitchen)
     (at tray33 kitchen)
     (at tray45 kitchen)
     (at_kitchen_bread bread37)
     (at_kitchen_content content78)
     (not_allergic_gluten child221)
     (waiting child221 table427)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child221)
    )
  )
)
