; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 546802

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child446 child42 - child
    bread149 bread380 - bread-portion
    content151 content419 - content-portion
    tray194 - tray
    table242 table177 table92 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at_kitchen_bread bread149)
     (at_kitchen_bread bread380)
     (at_kitchen_content content151)
     (at_kitchen_content content419)
     (no_gluten_bread bread380)
     (no_gluten_content content151)
     (allergic_gluten child42)
     (not_allergic_gluten child446)
     (waiting child446 table92)
     (waiting child42 table92)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child446)
     (served child42)
    )
  )
)
