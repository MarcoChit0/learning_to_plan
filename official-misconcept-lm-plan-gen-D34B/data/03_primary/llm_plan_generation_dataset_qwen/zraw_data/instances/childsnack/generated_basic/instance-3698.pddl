; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 322441

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child98 - child
    bread333 bread247 - bread-portion
    content66 content11 - content-portion
    tray411 tray418 - tray
    table254 table401 table388 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray411 kitchen)
     (at tray418 kitchen)
     (at_kitchen_bread bread333)
     (at_kitchen_bread bread247)
     (at_kitchen_content content66)
     (at_kitchen_content content11)
     (no_gluten_bread bread333)
     (no_gluten_content content11)
     (allergic_gluten child109)
     (not_allergic_gluten child98)
     (waiting child109 table401)
     (waiting child98 table388)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child109)
     (served child98)
    )
  )
)
