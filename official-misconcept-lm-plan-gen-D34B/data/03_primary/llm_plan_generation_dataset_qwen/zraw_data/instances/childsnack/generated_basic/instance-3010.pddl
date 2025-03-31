; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 338475

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child377 child202 - child
    bread201 bread58 - bread-portion
    content210 content5 - content-portion
    tray57 tray178 tray122 tray150 - tray
    table481 table288 table190 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray57 kitchen)
     (at tray178 kitchen)
     (at tray122 kitchen)
     (at tray150 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_bread bread58)
     (at_kitchen_content content210)
     (at_kitchen_content content5)
     (not_allergic_gluten child377)
     (not_allergic_gluten child202)
     (waiting child377 table288)
     (waiting child202 table288)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child377)
     (served child202)
    )
  )
)
