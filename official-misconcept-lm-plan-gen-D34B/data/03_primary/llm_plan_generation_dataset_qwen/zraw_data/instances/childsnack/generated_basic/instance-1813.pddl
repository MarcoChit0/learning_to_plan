; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 349609

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child255 child275 - child
    bread278 bread139 - bread-portion
    content388 content170 - content-portion
    tray160 - tray
    table90 table336 table186 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_bread bread139)
     (at_kitchen_content content388)
     (at_kitchen_content content170)
     (no_gluten_bread bread139)
     (no_gluten_content content388)
     (allergic_gluten child255)
     (not_allergic_gluten child275)
     (waiting child255 table336)
     (waiting child275 table90)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child255)
     (served child275)
    )
  )
)
