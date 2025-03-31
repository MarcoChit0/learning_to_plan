; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 921007

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child315 child403 child456 - child
    bread197 bread154 bread9 - bread-portion
    content356 content84 content388 - content-portion
    tray266 - tray
    table33 table344 table140 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray266 kitchen)
     (at_kitchen_bread bread197)
     (at_kitchen_bread bread154)
     (at_kitchen_bread bread9)
     (at_kitchen_content content356)
     (at_kitchen_content content84)
     (at_kitchen_content content388)
     (not_allergic_gluten child403)
     (not_allergic_gluten child315)
     (not_allergic_gluten child456)
     (waiting child315 table140)
     (waiting child403 table344)
     (waiting child456 table140)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child315)
     (served child403)
     (served child456)
    )
  )
)
