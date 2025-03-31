; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 459905

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 child101 - child
    bread234 bread285 - bread-portion
    content346 content305 - content-portion
    tray264 - tray
    table373 table430 table110 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray264 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_bread bread285)
     (at_kitchen_content content346)
     (at_kitchen_content content305)
     (no_gluten_bread bread285)
     (no_gluten_content content346)
     (allergic_gluten child101)
     (not_allergic_gluten child279)
     (waiting child279 table110)
     (waiting child101 table373)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child279)
     (served child101)
    )
  )
)
