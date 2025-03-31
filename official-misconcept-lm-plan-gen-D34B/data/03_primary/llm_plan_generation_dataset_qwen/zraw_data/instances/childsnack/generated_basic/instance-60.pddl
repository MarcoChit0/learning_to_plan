; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 680399

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 child209 - child
    bread335 bread444 - bread-portion
    content189 content443 - content-portion
    tray189 tray239 - tray
    table231 table82 table196 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread335)
     (at_kitchen_bread bread444)
     (at_kitchen_content content189)
     (at_kitchen_content content443)
     (not_allergic_gluten child295)
     (not_allergic_gluten child209)
     (waiting child295 table196)
     (waiting child209 table231)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child295)
     (served child209)
    )
  )
)
