; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 950416

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 child329 - child
    bread135 bread157 - bread-portion
    content131 content150 - content-portion
    tray318 tray150 tray395 - tray
    table237 table432 table396 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at tray150 kitchen)
     (at tray395 kitchen)
     (at_kitchen_bread bread135)
     (at_kitchen_bread bread157)
     (at_kitchen_content content131)
     (at_kitchen_content content150)
     (no_gluten_bread bread135)
     (no_gluten_content content150)
     (allergic_gluten child329)
     (not_allergic_gluten child333)
     (waiting child333 table396)
     (waiting child329 table396)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child333)
     (served child329)
    )
  )
)
