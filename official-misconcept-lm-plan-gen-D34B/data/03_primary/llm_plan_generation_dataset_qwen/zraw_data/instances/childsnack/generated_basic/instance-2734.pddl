; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 371157

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 child446 - child
    bread194 bread59 - bread-portion
    content115 content433 - content-portion
    tray182 - tray
    table8 table116 table272 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at_kitchen_bread bread194)
     (at_kitchen_bread bread59)
     (at_kitchen_content content115)
     (at_kitchen_content content433)
     (no_gluten_bread bread59)
     (no_gluten_content content115)
     (allergic_gluten child446)
     (not_allergic_gluten child200)
     (waiting child200 table116)
     (waiting child446 table116)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child200)
     (served child446)
    )
  )
)
