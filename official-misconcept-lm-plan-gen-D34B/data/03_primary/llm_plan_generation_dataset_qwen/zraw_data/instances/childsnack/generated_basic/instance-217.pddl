; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 270969

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child140 child183 - child
    bread114 bread204 - bread-portion
    content226 content17 - content-portion
    tray225 tray302 tray355 - tray
    table7 table428 table144 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray225 kitchen)
     (at tray302 kitchen)
     (at tray355 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_bread bread204)
     (at_kitchen_content content226)
     (at_kitchen_content content17)
     (not_allergic_gluten child183)
     (not_allergic_gluten child140)
     (waiting child140 table7)
     (waiting child183 table144)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child140)
     (served child183)
    )
  )
)
