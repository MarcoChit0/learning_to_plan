; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 388826

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 child381 child4 - child
    bread176 bread423 bread152 - bread-portion
    content229 content259 content48 - content-portion
    tray163 - tray
    table199 table314 table99 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray163 kitchen)
     (at_kitchen_bread bread176)
     (at_kitchen_bread bread423)
     (at_kitchen_bread bread152)
     (at_kitchen_content content229)
     (at_kitchen_content content259)
     (at_kitchen_content content48)
     (not_allergic_gluten child4)
     (not_allergic_gluten child381)
     (not_allergic_gluten child187)
     (waiting child187 table314)
     (waiting child381 table199)
     (waiting child4 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child187)
     (served child381)
     (served child4)
    )
  )
)
