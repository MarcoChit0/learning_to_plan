; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 150636

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child32 child85 - child
    bread415 bread102 - bread-portion
    content327 content92 - content-portion
    tray434 tray138 - tray
    table193 table58 table169 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at tray138 kitchen)
     (at_kitchen_bread bread415)
     (at_kitchen_bread bread102)
     (at_kitchen_content content327)
     (at_kitchen_content content92)
     (not_allergic_gluten child85)
     (not_allergic_gluten child32)
     (waiting child32 table58)
     (waiting child85 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child32)
     (served child85)
    )
  )
)
