; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 928436

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child263 child334 - child
    bread138 bread105 - bread-portion
    content0 content43 - content-portion
    tray460 tray310 - tray
    table317 table320 table31 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray460 kitchen)
     (at tray310 kitchen)
     (at_kitchen_bread bread138)
     (at_kitchen_bread bread105)
     (at_kitchen_content content0)
     (at_kitchen_content content43)
     (not_allergic_gluten child334)
     (not_allergic_gluten child263)
     (waiting child263 table31)
     (waiting child334 table317)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child263)
     (served child334)
    )
  )
)
