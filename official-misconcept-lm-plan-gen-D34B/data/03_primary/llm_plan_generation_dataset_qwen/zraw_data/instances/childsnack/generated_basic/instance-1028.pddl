; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 941870

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 child156 child226 - child
    bread154 bread443 bread116 - bread-portion
    content359 content287 content48 - content-portion
    tray302 - tray
    table411 table155 table219 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at_kitchen_bread bread154)
     (at_kitchen_bread bread443)
     (at_kitchen_bread bread116)
     (at_kitchen_content content359)
     (at_kitchen_content content287)
     (at_kitchen_content content48)
     (not_allergic_gluten child136)
     (not_allergic_gluten child226)
     (not_allergic_gluten child156)
     (waiting child136 table155)
     (waiting child156 table411)
     (waiting child226 table219)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child136)
     (served child156)
     (served child226)
    )
  )
)
