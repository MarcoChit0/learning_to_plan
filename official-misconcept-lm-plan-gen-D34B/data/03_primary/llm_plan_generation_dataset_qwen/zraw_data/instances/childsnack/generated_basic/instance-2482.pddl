; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 131757

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child90 child227 - child
    bread278 bread43 - bread-portion
    content487 content115 - content-portion
    tray402 tray348 - tray
    table347 table397 table126 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray402 kitchen)
     (at tray348 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_bread bread43)
     (at_kitchen_content content487)
     (at_kitchen_content content115)
     (not_allergic_gluten child227)
     (not_allergic_gluten child90)
     (waiting child90 table126)
     (waiting child227 table347)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child90)
     (served child227)
    )
  )
)
