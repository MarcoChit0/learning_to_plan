; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 153335

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 child158 - child
    bread40 bread251 - bread-portion
    content228 content412 - content-portion
    tray272 tray200 tray444 - tray
    table417 table338 table273 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray272 kitchen)
     (at tray200 kitchen)
     (at tray444 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_bread bread251)
     (at_kitchen_content content228)
     (at_kitchen_content content412)
     (not_allergic_gluten child158)
     (not_allergic_gluten child246)
     (waiting child246 table273)
     (waiting child158 table417)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child246)
     (served child158)
    )
  )
)
